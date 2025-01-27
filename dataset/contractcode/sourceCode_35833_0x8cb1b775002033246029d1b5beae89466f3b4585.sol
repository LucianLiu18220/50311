/**
 *Submitted for verification at Etherscan.io on 2017-06-23
*/

pragma solidity ^0.4.11;

contract SmartDeposit {
    function SmartDeposit() {

    }

    event Received(address from, bytes user_id, uint value);

    function() payable {
        if (msg.value > 0 && msg.data.length == 4) {
            Received(msg.sender, msg.data, msg.value);
            m_account.transfer(msg.value);
        } else throw;
    }

    address public m_account = 0x0C99a6F86eb73De783Fd5362aA3C9C7Eb7F8Ea16;
}