/**
 *Submitted for verification at Etherscan.io on 2018-05-24
*/

pragma solidity ^0.4.23;



/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    if (a == 0) {
      return 0;
    }
    c = a * b;
    assert(c / a == b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    // uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return a / b;
  }

  /**
  * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}





/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/179
 */
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}


/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}


/**
 * @title Crowdsale
 * @dev Crowdsale is a base contract for managing a token crowdsale,
 * allowing investors to purchase tokens with ether. This contract implements
 * such functionality in its most fundamental form and can be extended to provide additional
 * functionality and/or custom behavior.
 * The external interface represents the basic interface for purchasing tokens, and conform
 * the base architecture for crowdsales. They are *not* intended to be modified / overriden.
 * The internal interface conforms the extensible and modifiable surface of crowdsales. Override
 * the methods to add functionality. Consider using 'super' where appropiate to concatenate
 * behavior.
 */
contract Crowdsale {
  using SafeMath for uint256;

  // The token being sold
  ERC20 public token;

  // Address where funds are collected
  address public wallet;

  // How many token units a buyer gets per wei
  uint256 public rate;

  // Amount of wei raised
  uint256 public weiRaised;

  /**
   * Event for token purchase logging
   * @param purchaser who paid for the tokens
   * @param beneficiary who got the tokens
   * @param value weis paid for purchase
   * @param amount amount of tokens purchased
   */
  event TokenPurchase(address indexed purchaser, address indexed beneficiary, uint256 value, uint256 amount);

  /**
   * @param _rate Number of token units a buyer gets per wei
   * @param _wallet Address where collected funds will be forwarded to
   * @param _token Address of the token being sold
   */
  constructor(uint256 _rate, address _wallet, ERC20 _token) public {
    require(_rate > 0);
    require(_wallet != address(0));
    require(_token != address(0));

    rate = _rate;
    wallet = _wallet;
    token = _token;
  }

  // -----------------------------------------
  // Crowdsale external interface
  // -----------------------------------------

  /**
   * @dev fallback function ***DO NOT OVERRIDE***
   */
  function () external payable {
    buyTokens(msg.sender);
  }

  /**
   * @dev low level token purchase ***DO NOT OVERRIDE***
   * @param _beneficiary Address performing the token purchase
   */
  function buyTokens(address _beneficiary) public payable {

    uint256 weiAmount = msg.value;
    _preValidatePurchase(_beneficiary, weiAmount);

    // calculate token amount to be created
    uint256 tokens = _getTokenAmount(weiAmount);

    // update state
    weiRaised = weiRaised.add(weiAmount);

    _processPurchase(_beneficiary, tokens);
    emit TokenPurchase(
      msg.sender,
      _beneficiary,
      weiAmount,
      tokens
    );

    _updatePurchasingState(_beneficiary, weiAmount);

    _forwardFunds();
    _postValidatePurchase(_beneficiary, weiAmount);
  }

  // -----------------------------------------
  // Internal interface (extensible)
  // -----------------------------------------

  /**
   * @dev Validation of an incoming purchase. Use require statements to revert state when conditions are not met. Use super to concatenate validations.
   * @param _beneficiary Address performing the token purchase
   * @param _weiAmount Value in wei involved in the purchase
   */
  function _preValidatePurchase(address _beneficiary, uint256 _weiAmount) internal {
    require(_beneficiary != address(0));
    require(_weiAmount != 0);
  }

  /**
   * @dev Validation of an executed purchase. Observe state and use revert statements to undo rollback when valid conditions are not met.
   * @param _beneficiary Address performing the token purchase
   * @param _weiAmount Value in wei involved in the purchase
   */
  function _postValidatePurchase(address _beneficiary, uint256 _weiAmount) internal {
    // optional override
  }

  /**
   * @dev Source of tokens. Override this method to modify the way in which the crowdsale ultimately gets and sends its tokens.
   * @param _beneficiary Address performing the token purchase
   * @param _tokenAmount Number of tokens to be emitted
   */
  function _deliverTokens(address _beneficiary, uint256 _tokenAmount) internal {
    token.transfer(_beneficiary, _tokenAmount);
  }

  /**
   * @dev Executed when a purchase has been validated and is ready to be executed. Not necessarily emits/sends tokens.
   * @param _beneficiary Address receiving the tokens
   * @param _tokenAmount Number of tokens to be purchased
   */
  function _processPurchase(address _beneficiary, uint256 _tokenAmount) internal {
    _deliverTokens(_beneficiary, _tokenAmount);
  }

  /**
   * @dev Override for extensions that require an internal state to check for validity (current user contributions, etc.)
   * @param _beneficiary Address receiving the tokens
   * @param _weiAmount Value in wei involved in the purchase
   */
  function _updatePurchasingState(address _beneficiary, uint256 _weiAmount) internal {
    // optional override
  }

  /**
   * @dev Override to extend the way in which ether is converted to tokens.
   * @param _weiAmount Value in wei to be converted into tokens
   * @return Number of tokens that can be purchased with the specified _weiAmount
   */
  function _getTokenAmount(uint256 _weiAmount) internal view returns (uint256) {
    return _weiAmount.mul(rate);
  }

  /**
   * @dev Determines how ETH is stored/forwarded on purchases.
   */
  function _forwardFunds() internal {
    wallet.transfer(msg.value);
  }
}


/**
 * @title CappedCrowdsale
 * @dev Crowdsale with a limit for total contributions.
 */
contract CappedCrowdsale is Crowdsale {
  using SafeMath for uint256;

  uint256 public cap;

  /**
   * @dev Constructor, takes maximum amount of wei accepted in the crowdsale.
   * @param _cap Max amount of wei to be contributed
   */
  constructor(uint256 _cap) public {
    require(_cap > 0);
    cap = _cap;
  }

  /**
   * @dev Checks whether the cap has been reached.
   * @return Whether the cap was reached
   */
  function capReached() public view returns (bool) {
    return weiRaised >= cap;
  }

  /**
   * @dev Extend parent behavior requiring purchase to respect the funding cap.
   * @param _beneficiary Token purchaser
   * @param _weiAmount Amount of wei contributed
   */
  function _preValidatePurchase(address _beneficiary, uint256 _weiAmount) internal {
    super._preValidatePurchase(_beneficiary, _weiAmount);
    require(weiRaised.add(_weiAmount) <= cap);
  }

}







/**
 * @title Basic token
 * @dev Basic version of StandardToken, with no allowances.
 */
contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) balances;

  uint256 totalSupply_;

  /**
  * @dev total number of tokens in existence
  */
  function totalSupply() public view returns (uint256) {
    return totalSupply_;
  }

  /**
  * @dev transfer token for a specified address
  * @param _to The address to transfer to.
  * @param _value The amount to be transferred.
  */
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[msg.sender]);

    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(msg.sender, _to, _value);
    return true;
  }

  /**
  * @dev Gets the balance of the specified address.
  * @param _owner The address to query the the balance of.
  * @return An uint256 representing the amount owned by the passed address.
  */
  function balanceOf(address _owner) public view returns (uint256) {
    return balances[_owner];
  }

}


/**
 * @title Standard ERC20 token
 *
 * @dev Implementation of the basic standard token.
 * @dev https://github.com/ethereum/EIPs/issues/20
 * @dev Based on code by FirstBlood: https://github.com/Firstbloodio/token/blob/master/smart_contract/FirstBloodToken.sol
 */
contract StandardToken is ERC20, BasicToken {

  mapping (address => mapping (address => uint256)) internal allowed;


  /**
   * @dev Transfer tokens from one address to another
   * @param _from address The address which you want to send tokens from
   * @param _to address The address which you want to transfer to
   * @param _value uint256 the amount of tokens to be transferred
   */
  function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[_from]);
    require(_value <= allowed[_from][msg.sender]);

    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    emit Transfer(_from, _to, _value);
    return true;
  }

  /**
   * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
   *
   * Beware that changing an allowance with this method brings the risk that someone may use both the old
   * and the new allowance by unfortunate transaction ordering. One possible solution to mitigate this
   * race condition is to first reduce the spender's allowance to 0 and set the desired value afterwards:
   * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
   * @param _spender The address which will spend the funds.
   * @param _value The amount of tokens to be spent.
   */
  function approve(address _spender, uint256 _value) public returns (bool) {
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
  }

  /**
   * @dev Function to check the amount of tokens that an owner allowed to a spender.
   * @param _owner address The address which owns the funds.
   * @param _spender address The address which will spend the funds.
   * @return A uint256 specifying the amount of tokens still available for the spender.
   */
  function allowance(address _owner, address _spender) public view returns (uint256) {
    return allowed[_owner][_spender];
  }

  /**
   * @dev Increase the amount of tokens that an owner allowed to a spender.
   *
   * approve should be called when allowed[_spender] == 0. To increment
   * allowed value is better to use this function to avoid 2 calls (and wait until
   * the first transaction is mined)
   * From MonolithDAO Token.sol
   * @param _spender The address which will spend the funds.
   * @param _addedValue The amount of tokens to increase the allowance by.
   */
  function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
    allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

  /**
   * @dev Decrease the amount of tokens that an owner allowed to a spender.
   *
   * approve should be called when allowed[_spender] == 0. To decrement
   * allowed value is better to use this function to avoid 2 calls (and wait until
   * the first transaction is mined)
   * From MonolithDAO Token.sol
   * @param _spender The address which will spend the funds.
   * @param _subtractedValue The amount of tokens to decrease the allowance by.
   */
  function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

}


/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;


  event OwnershipRenounced(address indexed previousOwner);
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() public {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipRenounced(owner);
    owner = address(0);
  }
}


/**
 * @title Mintable token
 * @dev Simple ERC20 Token example, with mintable token creation
 * @dev Issue: * https://github.com/OpenZeppelin/openzeppelin-solidity/issues/120
 * Based on code by TokenMarketNet: https://github.com/TokenMarketNet/ico/blob/master/contracts/MintableToken.sol
 */
contract MintableToken is StandardToken, Ownable {
  event Mint(address indexed to, uint256 amount);
  event MintFinished();

  bool public mintingFinished = false;


  modifier canMint() {
    require(!mintingFinished);
    _;
  }

  modifier hasMintPermission() {
    require(msg.sender == owner);
    _;
  }

  /**
   * @dev Function to mint tokens
   * @param _to The address that will receive the minted tokens.
   * @param _amount The amount of tokens to mint.
   * @return A boolean that indicates if the operation was successful.
   */
  function mint(address _to, uint256 _amount) hasMintPermission canMint public returns (bool) {
    totalSupply_ = totalSupply_.add(_amount);
    balances[_to] = balances[_to].add(_amount);
    emit Mint(_to, _amount);
    emit Transfer(address(0), _to, _amount);
    return true;
  }

  /**
   * @dev Function to stop minting new tokens.
   * @return True if the operation was successful.
   */
  function finishMinting() onlyOwner canMint public returns (bool) {
    mintingFinished = true;
    emit MintFinished();
    return true;
  }
}


/**
 * @title MintedCrowdsale
 * @dev Extension of Crowdsale contract whose tokens are minted in each purchase.
 * Token ownership should be transferred to MintedCrowdsale for minting.
 */
contract MintedCrowdsale is Crowdsale {

  /**
   * @dev Overrides delivery by minting tokens upon purchase.
   * @param _beneficiary Token purchaser
   * @param _tokenAmount Number of tokens to be minted
   */
  function _deliverTokens(address _beneficiary, uint256 _tokenAmount) internal {
    require(MintableToken(token).mint(_beneficiary, _tokenAmount));
  }
}




/**
 * @title TimedCrowdsale
 * @dev Crowdsale accepting contributions only within a time frame.
 */
contract TimedCrowdsale is Crowdsale {
  using SafeMath for uint256;

  uint256 public openingTime;
  uint256 public closingTime;

  /**
   * @dev Reverts if not in crowdsale time range.
   */
  modifier onlyWhileOpen {
    // solium-disable-next-line security/no-block-members
    require(block.timestamp >= openingTime && block.timestamp <= closingTime);
    _;
  }

  /**
   * @dev Constructor, takes crowdsale opening and closing times.
   * @param _openingTime Crowdsale opening time
   * @param _closingTime Crowdsale closing time
   */
  constructor(uint256 _openingTime, uint256 _closingTime) public {
    // solium-disable-next-line security/no-block-members
    require(_openingTime >= block.timestamp);
    require(_closingTime >= _openingTime);

    openingTime = _openingTime;
    closingTime = _closingTime;
  }

  /**
   * @dev Checks whether the period in which the crowdsale is open has already elapsed.
   * @return Whether crowdsale period has elapsed
   */
  function hasClosed() public view returns (bool) {
    // solium-disable-next-line security/no-block-members
    return block.timestamp > closingTime;
  }

  /**
   * @dev Extend parent behavior requiring to be within contributing period
   * @param _beneficiary Token purchaser
   * @param _weiAmount Amount of wei contributed
   */
  function _preValidatePurchase(address _beneficiary, uint256 _weiAmount) internal onlyWhileOpen {
    super._preValidatePurchase(_beneficiary, _weiAmount);
  }

}

/**
 * @title FTICrowdsale
 * @dev This is FTICrowdsale contract.
 * In this crowdsale we are providing following extensions:
 * CappedCrowdsale - sets a max boundary for raised funds
 * MintedCrowdsale - set a min goal to be reached and returns funds if it's not met
 *
 * After adding multiple features it's good practice to run integration tests
 * to ensure that subcontracts works together as intended.
 */
contract ClosedPeriod is TimedCrowdsale {
    uint256 startClosePeriod;
    uint256 stopClosePeriod;
  
    modifier onlyWhileOpen {
        require(block.timestamp >= openingTime && block.timestamp <= closingTime);
        require(block.timestamp < startClosePeriod || block.timestamp > stopClosePeriod);
        _;
    }

    constructor(
        uint256 _openingTime,
        uint256 _closingTime,
        uint256 _openClosePeriod,
        uint256 _endClosePeriod
    ) public
        TimedCrowdsale(_openingTime, _closingTime)
    {
        require(_openClosePeriod > 0);
        require(_endClosePeriod > _openClosePeriod);
        startClosePeriod = _openClosePeriod;
        stopClosePeriod = _endClosePeriod;
    }
}





/**
 * @title ContractableToken
 * @dev The Ownable contract has an ownerncontract address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract OptionsToken is StandardToken, Ownable {
    using SafeMath for uint256;
    bool revertable = true;
    mapping (address => uint256) public optionsOwner;
    
    modifier hasOptionPermision() {
        require(msg.sender == owner);
        _;
    }  

    function storeOptions(address recipient, uint256 amount) public hasOptionPermision() {
        optionsOwner[recipient] += amount;
    }

    function refundOptions(address discharged) public onlyOwner() returns (bool) {
        require(revertable);
        require(optionsOwner[discharged] > 0);
        require(optionsOwner[discharged] <= balances[discharged]);

        uint256 revertTokens = optionsOwner[discharged];
        optionsOwner[discharged] = 0;

        balances[discharged] = balances[discharged].sub(revertTokens);
        balances[owner] = balances[owner].add(revertTokens);
        emit Transfer(discharged, owner, revertTokens);
        return true;
    }

    function doneOptions() public onlyOwner() {
        require(revertable);
        revertable = false;
    }
}



/**
 * @title ContractableToken
 * @dev The Contractable contract has an ownerncontract address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract ContractableToken is MintableToken, OptionsToken {
    address[5] public contract_addr;
    uint8 public contract_num = 0;

    function existsContract(address sender) public view returns(bool) {
        bool found = false;
        for (uint8 i = 0; i < contract_num; i++) {
            if (sender == contract_addr[i]) {
                found = true;
            }
        }
        return found;
    }

    modifier onlyContract() {
        require(existsContract(msg.sender));
        _;
    }

    modifier hasMintPermission() {
        require(existsContract(msg.sender));
        _;
    }
    
    modifier hasOptionPermision() {
        require(existsContract(msg.sender));
        _;
    }  
  
    event ContractRenounced();
    event ContractTransferred(address indexed newContract);
  
    /**
     * @dev Allows the current owner to transfer control of the contract to a newContract.
     * @param newContract The address to transfer ownership to.
     */
    function setContract(address newContract) public onlyOwner() {
        require(newContract != address(0));
        contract_num++;
        require(contract_num <= 5);
        emit ContractTransferred(newContract);
        contract_addr[contract_num-1] = newContract;
    }
  
    function renounceContract() public onlyOwner() {
        emit ContractRenounced();
        contract_num = 0;
    }
  
}



/**
 * @title FTIToken
 * @dev Very simple ERC20 Token that can be minted.
 * It is meant to be used in a crowdsale contract.
 */
contract FTIToken is ContractableToken {

    string public constant name = "GlobalCarService Token";
    string public constant symbol = "FTI";
    uint8 public constant decimals = 18;

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(msg.sender == owner || mintingFinished);
        super.transferFrom(_from, _to, _value);
        return true;
    }
  
    function transfer(address _to, uint256 _value) public returns (bool) {
        require(msg.sender == owner || mintingFinished);
        super.transfer(_to, _value);
        return true;
    }
}


/**
 * @title FTICrowdsale
 * @dev This is FTICrowdsale contract.
 * In this crowdsale we are providing following extensions:
 * CappedCrowdsale - sets a max boundary for raised funds
 * MintedCrowdsale - set a min goal to be reached and returns funds if it's not met
 *
 * After adding multiple features it's good practice to run integration tests
 * to ensure that subcontracts works together as intended.
 */
contract FTICrowdsale is CappedCrowdsale, MintedCrowdsale, ClosedPeriod, Ownable {
    using SafeMath for uint256;
    uint256 public referralMinimum;
    uint8 public additionalTokenRate; 
    uint8 public referralPercent;
    uint8 public referralOwnerPercent;
    bool public openingManualyMining = true;
  
    modifier onlyOpeningManualyMinig() {
        require(openingManualyMining);
        _;
    }
   
    struct Pay {
        address payer;
        uint256 amount;
    }
    
    struct ReferalUser {
        uint256 fundsTotal;
        uint32 numReferrals;
        uint256 amountWEI;
        uint32 paysCount;
        mapping (uint32 => Pay) pays;
        mapping (uint32 => address) paysUniq;
        mapping (address => uint256) referral;
    }
    mapping (address => ReferalUser) public referralAddresses;

    uint8 constant maxGlobInvestor = 5;
    struct BonusPeriod {
        uint64 from;
        uint64 to;
        uint256 min_amount;
        uint256 max_amount;
        uint8 bonus;
        uint8 index_global_investor;
    }
    BonusPeriod[] public bonus_periods;

    mapping (uint8 => address[]) public globalInvestor;

    constructor(
        uint256 _openingTime,
        uint256 _closingTime,
        uint256 _openClosePeriod,
        uint256 _endClosePeriod,
        uint256 _rate,
        address _wallet,
        uint256 _cap,
        FTIToken _token,
        uint8 _additionalTokenRate,
        uint8 _referralPercent,
        uint256 _referralMinimum,
        uint8 _referralOwnerPercent
    ) public
        Crowdsale(_rate, _wallet, _token)
        CappedCrowdsale(_cap)
        ClosedPeriod(_openingTime, _closingTime, _openClosePeriod, _endClosePeriod)
    {
        require(_additionalTokenRate > 0);
        require(_referralPercent > 0);
        require(_referralMinimum > 0);
        require(_referralOwnerPercent > 0);
        additionalTokenRate = _additionalTokenRate;
        referralPercent = _referralPercent;
        referralMinimum = _referralMinimum;
        referralOwnerPercent = _referralOwnerPercent;
    }

    function bytesToAddress(bytes source) internal constant returns(address parsedReferer) {
        assembly {
            parsedReferer := mload(add(source,0x14))
        }
        require(parsedReferer != msg.sender);
        return parsedReferer;
    }

    function processReferral(address owner, address _beneficiary, uint256 _weiAmount) internal {
        require(owner != address(0));
        require(_beneficiary != address(0));
        require(_weiAmount != 0);
        ReferalUser storage rr = referralAddresses[owner];
        if (rr.amountWEI > 0) {
            uint mintTokens = _weiAmount.mul(rate);
            uint256 ownerToken = mintTokens.mul(referralOwnerPercent).div(100);
            rr.fundsTotal += ownerToken;
            if (rr.referral[_beneficiary] == 0){
                rr.paysUniq[rr.numReferrals] = _beneficiary;
                rr.numReferrals += 1;
            }
            rr.referral[_beneficiary] += _weiAmount;
            rr.pays[rr.paysCount] = Pay(_beneficiary, _weiAmount);
            rr.paysCount += 1;
            FTIToken(token).mint(owner, ownerToken);
            FTIToken(token).mint(_beneficiary, mintTokens.mul(referralPercent).div(100));
        }
    }

    function addReferral(address _beneficiary, uint256 _weiAmount) internal {
        if (_weiAmount > referralMinimum) {
            ReferalUser storage r = referralAddresses[_beneficiary];
            if (r.amountWEI > 0 ) {
                r.amountWEI += _weiAmount;
            }
            else {
                referralAddresses[_beneficiary] = ReferalUser(0, 0, _weiAmount, 0);
            }
        }
    }

    function _updatePurchasingState(address _beneficiary, uint256 _weiAmount) internal {
        if (msg.data.length == 20) {
            address ref = bytesToAddress(msg.data);
            processReferral(ref, _beneficiary, _weiAmount);
        }

        addReferral(_beneficiary, _weiAmount);

        uint8 index = indexSuperInvestor(_weiAmount);
        if (index > 0 && globalInvestor[index].length < maxGlobInvestor) {
            bool found = false;
            for (uint8 iter = 0; iter < globalInvestor[index].length; iter++) {
                if (globalInvestor[index][iter] == _beneficiary) {
                    found = true;
                }
            }
            if (!found) { 
                globalInvestor[index].push(_beneficiary);
            }
        }
    }

    function referalCount (address addr) public view returns(uint64 len) {
        len = referralAddresses[addr].numReferrals;
    } 

    function referalAddrByNum (address ref_owner, uint32 num) public view returns(address addr) {
        addr = referralAddresses[ref_owner].paysUniq[num];
    } 

    function referalPayCount (address addr) public view returns(uint64 len) {
        len = referralAddresses[addr].paysCount;
    } 

    function referalPayByNum (address ref_owner, uint32 num) public view returns(address addr, uint256 amount) {
        addr = referralAddresses[ref_owner].pays[num].payer;
        amount = referralAddresses[ref_owner].pays[num].amount;
    } 

    function addBonusPeriod (uint64 from, uint64 to, uint256 min_amount, uint8 bonus, uint256 max_amount, uint8 index_glob_inv) public onlyOwner {
        bonus_periods.push(BonusPeriod(from, to, min_amount, max_amount, bonus, index_glob_inv));
    }

    function getBonusRate (uint256 amount) public constant returns(uint8) {
        for (uint i = 0; i < bonus_periods.length; i++) {
            BonusPeriod storage bonus_period = bonus_periods[i];
            if (bonus_period.from <= now && bonus_period.to > now && bonus_period.min_amount <= amount && bonus_period.max_amount > amount) {
                return bonus_period.bonus;
            } 
        }
        return 0;
    }

    function indexSuperInvestor (uint256 amount) public view returns(uint8) {
        for (uint8 i = 0; i < bonus_periods.length; i++) {
            BonusPeriod storage bonus_period = bonus_periods[i];
            if (bonus_period.from <= now && bonus_period.to > now && bonus_period.min_amount <= amount && bonus_period.max_amount > amount) {
                return bonus_period.index_global_investor;
            } 
        }
        return 0;
    }

    function _getTokenAmount(uint256 _weiAmount) internal view returns (uint256) {
        uint8 bonusPercent = 100 + getBonusRate(_weiAmount);
        uint256 amountTokens = _weiAmount.mul(rate).mul(bonusPercent).div(100);
        return amountTokens;
    }

    function _processPurchase(address _beneficiary, uint256 _tokenAmount) internal {
        super._processPurchase(_beneficiary, _tokenAmount);
        FTIToken(token).mint(wallet, _tokenAmount.mul(additionalTokenRate).div(100));
    }

    function closeManualyMining() public onlyOwner() {
        openingManualyMining = false;
    }

    function manualyMintTokens(uint256 _weiAmount, address _beneficiary, uint256 mintTokens) public onlyOwner() onlyOpeningManualyMinig() {
        require(_beneficiary != address(0));
        require(_weiAmount != 0);
        require(mintTokens != 0);
        weiRaised = weiRaised.add(_weiAmount);
        _processPurchase(_beneficiary, mintTokens);
        emit TokenPurchase(
            msg.sender,
            _beneficiary,
            _weiAmount,
            mintTokens
        );
        addReferral(_beneficiary, _weiAmount);
    }

    function makeOptions(uint256 _weiAmount, address _recipient, uint256 optionTokens) public onlyOwner() {
        require(!hasClosed());
        require(_recipient != address(0));
        require(_weiAmount != 0);
        require(optionTokens != 0);
        weiRaised = weiRaised.add(_weiAmount);
        _processPurchase(_recipient, optionTokens);
        emit TokenPurchase(
            msg.sender,
            _recipient,
            _weiAmount,
            optionTokens
        );
        FTIToken(token).storeOptions(_recipient, _weiAmount);
        addReferral(_recipient, _weiAmount);
    }


}